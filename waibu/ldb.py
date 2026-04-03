import plyvel
import json
import os
import sys
from datetime import datetime

def read_leveldb_data_detailed(db_directory, max_display=100):
    """
    详细读取 LevelDB 数据库数据
    Args:
        db_directory: LevelDB 数据库目录路径
        max_display: 最大显示记录数
    """
    
    print(f"🔍 正在打开 LevelDB 数据库...")
    print(f"📂 路径: {db_directory}")
    print("=" * 80)
    
    # 验证目录是否存在
    if not os.path.exists(db_directory):
        print(f"❌ 错误: 目录不存在")
        return None
    
    # 验证是否是 LevelDB 数据库
    required_files = ['CURRENT', 'MANIFEST', 'LOCK']
    existing_files = os.listdir(db_directory)
    
    print(f"📊 数据库文件检查:")
    for file in required_files:
        if file in existing_files:
            print(f"   ✅ {file}")
        else:
            for f in existing_files:
                if file in f:  # 部分匹配（如 MANIFEST-000001）
                    print(f"   ✅ {f}")
                    break
            else:
                print(f"   ❌ {file} (未找到)")
    
    try:
        # 打开数据库
        print(f"\n🔓 打开数据库中...")
        db = plyvel.DB(db_directory, create_if_missing=False)
        
        # 创建快照以确保一致性读取
        snapshot = db.snapshot()
        
        # 统计信息
        print(f"📈 正在统计记录...")
        total_count = 0
        key_lengths = []
        value_lengths = []
        
        # 第一次遍历：统计信息
        for key, value in snapshot:
            total_count += 1
            key_lengths.append(len(key))
            value_lengths.append(len(value))
        
        print(f"\n✅ 数据库信息:")
        print(f"   总记录数: {total_count:,}")
        if total_count > 0:
            print(f"   Key 平均长度: {sum(key_lengths)/len(key_lengths):.1f} 字节")
            print(f"   Value 平均长度: {sum(value_lengths)/len(value_lengths):.1f} 字节")
            print(f"   数据总大小: {sum(key_lengths) + sum(value_lengths):,} 字节")
        
        print(f"\n" + "=" * 80)
        print(f"📄 数据内容 (显示前 {min(max_display, total_count)} 条):")
        print("=" * 80)
        
        # 第二次遍历：显示数据
        display_count = 0
        data_records = []
        
        for key_bytes, value_bytes in snapshot:
            display_count += 1
            
            if display_count > max_display:
                print(f"\n... 还有 {total_count - max_display} 条记录未显示")
                break
            
            print(f"\n🔢 记录 #{display_count}")
            print(f"   {'─' * 40}")
            
            # 1. 解码 Key
            try:
                # 尝试多种编码
                try:
                    key_str = key_bytes.decode('utf-8')
                    key_encoding = 'UTF-8'
                except:
                    try:
                        key_str = key_bytes.decode('gbk')
                        key_encoding = 'GBK'
                    except:
                        try:
                            key_str = key_bytes.decode('latin-1')
                            key_encoding = 'Latin-1'
                        except:
                            key_str = key_bytes.hex()
                            key_encoding = 'HEX'
                
                print(f"   🔑 Key ({key_encoding}):")
                print(f"      {key_str}")
                print(f"      长度: {len(key_bytes)} 字节")
                
            except Exception as e:
                print(f"   🔑 Key (解码失败):")
                print(f"      原始数据: {key_bytes.hex()[:50]}...")
                print(f"      长度: {len(key_bytes)} 字节")
                key_str = key_bytes.hex()
            
            # 2. 解码 Value
            print(f"\n   📦 Value:")
            print(f"      长度: {len(value_bytes)} 字节")
            
            # 尝试多种解码方式
            decoded_success = False
            
            # 尝试作为 JSON 解码
            try:
                value_json = json.loads(value_bytes.decode('utf-8'))
                print(f"      🔍 格式: JSON")
                print(f"      📝 内容:")
                print(json.dumps(value_json, indent=6, ensure_ascii=False)[:200])
                if len(json.dumps(value_json)) > 200:
                    print(f"      ... (内容过长，已截断)")
                decoded_success = True
            except:
                pass
            
            # 尝试作为文本解码
            if not decoded_success:
                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                    try:
                        value_text = value_bytes.decode(encoding)
                        if len(value_text.strip()) > 0:
                            print(f"      🔍 格式: 文本 ({encoding})")
                            print(f"      📝 内容: {value_text[:150]}")
                            if len(value_text) > 150:
                                print(f"      ... (内容过长，已截断)")
                            decoded_success = True
                            break
                    except:
                        continue
            
            # 如果是二进制数据
            if not decoded_success:
                print(f"      🔍 格式: 二进制")
                
                # 检查常见格式
                if len(value_bytes) >= 4:
                    # 检查是否是整数
                    if len(value_bytes) == 4:
                        try:
                            int_value = int.from_bytes(value_bytes, 'little')
                            print(f"      📊 可能为32位整数: {int_value}")
                            decoded_success = True
                        except:
                            pass
                    
                    # 检查是否是浮点数
                    if len(value_bytes) == 8 and not decoded_success:
                        try:
                            import struct
                            float_value = struct.unpack('<d', value_bytes)[0]
                            print(f"      📊 可能为64位浮点数: {float_value}")
                            decoded_success = True
                        except:
                            pass
                
                if not decoded_success:
                    # 显示十六进制预览
                    hex_preview = value_bytes.hex()[:100]
                    print(f"      🔢 十六进制: {hex_preview}...")
                    
                    # 尝试提取可打印字符
                    printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in value_bytes[:50])
                    if any(c != '.' for c in printable):
                        print(f"      📝 可打印字符: {printable}")
            
            # 保存记录
            data_records.append({
                'index': display_count,
                'key_raw': key_bytes.hex(),
                'key_decoded': key_str if isinstance(key_str, str) else key_str.hex(),
                'value_raw': value_bytes.hex(),
                'value_length': len(value_bytes),
                'value_preview': value_bytes[:100].hex() if len(value_bytes) > 100 else value_bytes.hex()
            })
            
            # 每5条记录后暂停
            if display_count % 5 == 0 and display_count < max_display:
                input(f"\n   ⏸️  已显示 {display_count} 条，按 Enter 继续...")
                print(f"\n   {'─' * 40}")
        
        # 关闭资源
        snapshot.close()
        db.close()
        
        print(f"\n" + "=" * 80)
        print(f"✅ 读取完成!")
        print(f"📊 共读取 {display_count} 条记录")
        
        return data_records
        
    except Exception as e:
        print(f"\n❌ 打开数据库失败: {e}")
        print(f"\n💡 可能的原因:")
        print(f"   1. 这不是一个有效的 LevelDB 数据库目录")
        print(f"   2. 数据库文件损坏")
        print(f"   3. 缺少必要的文件 (CURRENT, MANIFEST, LOCK)")
        print(f"   4. 权限不足")
        return None

# 3. 导出数据功能
def export_leveldb_data(db_directory, output_format='json'):
    """导出 LevelDB 数据"""
    
    try:
        db = plyvel.DB(db_directory, create_if_missing=False)
        
        data = {}
        for key_bytes, value_bytes in db:
            # 解码 key
            try:
                key_str = key_bytes.decode('utf-8')
            except:
                key_str = key_bytes.hex()
            
            # 解码 value
            try:
                # 尝试 JSON
                value_json = json.loads(value_bytes.decode('utf-8'))
                data[key_str] = value_json
            except:
                # 尝试文本
                try:
                    value_text = value_bytes.decode('utf-8')
                    data[key_str] = value_text
                except:
                    # 保存为 base64
                    import base64
                    data[key_str] = {
                        '_format': 'binary',
                        '_size': len(value_bytes),
                        '_data': base64.b64encode(value_bytes).decode('ascii')
                    }
        
        db.close()
        
        # 导出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == 'json':
            output_file = f"leveldb_export_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ 已导出 JSON: {output_file}")
            print(f"   记录数: {len(data)}")
            
        elif output_format == 'csv':
            output_file = f"leveldb_export_{timestamp}.csv"
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Key', 'Value', 'Size'])
                for key, value in data.items():
                    if isinstance(value, dict) and '_format' in value:
                        writer.writerow([key, f"[Binary {value['_size']} bytes]", value['_size']])
                    else:
                        value_str = str(value)[:200]  # 截断长文本
                        writer.writerow([key, value_str, len(str(value))])
            print(f"✅ 已导出 CSV: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return None

# 4. 搜索功能
def search_in_leveldb(db_directory, search_term):
    """在 LevelDB 中搜索"""
    
    try:
        db = plyvel.DB(db_directory, create_if_missing=False)
        
        print(f"🔍 搜索: '{search_term}'")
        print("=" * 60)
        
        results = []
        for key_bytes, value_bytes in db:
            # 在 key 中搜索
            try:
                key_str = key_bytes.decode('utf-8', errors='ignore')
                if search_term.lower() in key_str.lower():
                    results.append(('key', key_str, value_bytes))
                    continue
            except:
                pass
            
            # 在 value 中搜索
            try:
                value_str = value_bytes.decode('utf-8', errors='ignore')
                if search_term.lower() in value_str.lower():
                    results.append(('value', key_bytes, value_bytes))
            except:
                # 尝试二进制搜索
                if search_term.encode('utf-8') in value_bytes:
                    results.append(('binary', key_bytes, value_bytes))
        
        db.close()
        
        if results:
            print(f"✅ 找到 {len(results)} 个结果:")
            print("-" * 60)
            for i, (match_type, key, value) in enumerate(results, 1):
                print(f"\n{i}. 匹配位置: {match_type}")
                
                # 显示 key
                try:
                    if isinstance(key, bytes):
                        key_display = key.decode('utf-8', errors='ignore')
                    else:
                        key_display = key
                    print(f"   🔑 Key: {key_display[:80]}...")
                except:
                    print(f"   🔑 Key (hex): {key.hex()[:40]}...")
                
                # 显示 value 预览
                try:
                    value_display = value.decode('utf-8', errors='ignore')
                    print(f"   📦 Value: {value_display[:150]}...")
                except:
                    print(f"   📦 Value (二进制, {len(value)} 字节)")
                    
        else:
            print(f"❌ 未找到包含 '{search_term}' 的记录")
        
        return results
        
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        return None

# 5. 主函数
def main():
    """主程序"""
    
    # 你的文件路径
    file_path = "/media/sdc/liujiayu/TDN-main/waibu/chakan.ldb"
    db_directory = os.path.dirname(file_path)  # 获取目录路径
    
    print("🔍 LevelDB 数据读取工具")
    print("=" * 60)
    
    # 检查目录
    if not os.path.exists(db_directory):
        print(f"❌ 目录不存在: {db_directory}")
        
        # 尝试父目录
        parent_dir = os.path.dirname(db_directory)
        print(f"💡 尝试父目录: {parent_dir}")
        if os.path.exists(parent_dir):
            db_directory = parent_dir
        else:
            return
    
    print(f"📂 数据库目录: {db_directory}")
    print(f"📄 目标文件: {os.path.basename(file_path)}")
    
    while True:
        print(f"\n" + "=" * 60)
        print("📋 菜单选项:")
        print("  1. 查看所有数据")
        print("  2. 查看前N条数据")
        print("  3. 搜索数据")
        print("  4. 导出数据")
        print("  5. 显示统计信息")
        print("  6. 退出")
        
        choice = input("\n请选择 (1-6): ").strip()
        
        if choice == '1':
            read_leveldb_data_detailed(db_directory, max_display=50)
            
        elif choice == '2':
            try:
                n = int(input("显示多少条记录? (默认50): ") or "50")
                read_leveldb_data_detailed(db_directory, max_display=n)
            except ValueError:
                print("❌ 请输入有效的数字")
                
        elif choice == '3':
            search_term = input("输入搜索关键词: ").strip()
            if search_term:
                search_in_leveldb(db_directory, search_term)
            else:
                print("❌ 搜索词不能为空")
                
        elif choice == '4':
            print("选择导出格式:")
            print("  1. JSON (推荐)")
            print("  2. CSV")
            format_choice = input("选择 (1-2): ").strip()
            if format_choice == '1':
                export_leveldb_data(db_directory, 'json')
            elif format_choice == '2':
                export_leveldb_data(db_directory, 'csv')
            else:
                print("❌ 无效选择")
                
        elif choice == '5':
            try:
                db = plyvel.DB(db_directory, create_if_missing=False)
                count = 0
                total_key_size = 0
                total_value_size = 0
                
                for key, value in db:
                    count += 1
                    total_key_size += len(key)
                    total_value_size += len(value)
                
                db.close()
                
                print(f"\n📊 数据库统计:")
                print(f"   总记录数: {count:,}")
                if count > 0:
                    print(f"   Key 总大小: {total_key_size:,} 字节")
                    print(f"   Value 总大小: {total_value_size:,} 字节")
                    print(f"   数据总大小: {total_key_size + total_value_size:,} 字节")
                    print(f"   Key 平均大小: {total_key_size/count:.1f} 字节")
                    print(f"   Value 平均大小: {total_value_size/count:.1f} 字节")
                    
            except Exception as e:
                print(f"❌ 获取统计失败: {e}")
                
        elif choice == '6':
            print("👋 再见!")
            break
            
        else:
            print("❌ 无效选择，请重试")

# 6. 直接运行（简单方式）
def quick_read():
    """快速读取数据"""
    db_directory = "/media/sdc/liujiayu/TDN-main/waibu/"
    
    try:
        db = plyvel.DB(db_directory, create_if_missing=False)
        
        print("🔍 快速预览数据 (前10条):")
        print("=" * 60)
        
        count = 0
        for key, value in db:
            count += 1
            if count > 10:
                break
                
            print(f"\n记录 #{count}:")
            
            # Key
            try:
                key_str = key.decode('utf-8')
                print(f"  Key: {key_str}")
            except:
                print(f"  Key (hex): {key.hex()[:40]}...")
            
            # Value
            try:
                value_str = value.decode('utf-8')
                if len(value_str) > 100:
                    print(f"  Value: {value_str[:100]}...")
                else:
                    print(f"  Value: {value_str}")
            except:
                print(f"  Value (二进制): {len(value)} 字节")
        
        db.close()
        print(f"\n✅ 共找到 {count} 条记录")
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        print(f"💡 尝试的路径: {db_directory}")

# 运行主程序
if __name__ == "__main__":
    # 直接运行快速读取
    quick_read()
    
    # 或者运行完整菜单
    # main()